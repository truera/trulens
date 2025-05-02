import type { Meta, StoryObj } from '@storybook/react';
import LabelAndValue from './LabelAndValue';
import { Typography } from '@mui/material';

type Story = StoryObj<typeof LabelAndValue>;

const meta: Meta<typeof LabelAndValue> = {
  title: 'Components/LabelAndValue',
  component: LabelAndValue,
  args: {
    label: 'Field Label',
    value: 'Field Value',
  },
};

export default meta;

export const Default: Story = {};

export const WithLongValue: Story = {
  args: {
    value: 'This is a longer value that may wrap to multiple lines depending on the container width.',
  },
};

export const WithCustomAlignment: Story = {
  args: {
    align: 'center',
  },
};

export const WithReactNodeValue: Story = {
  args: {
    value: <Typography sx={{ border: '1px solid red', padding: 1 }}>Custom formatted value</Typography>,
  },
};
